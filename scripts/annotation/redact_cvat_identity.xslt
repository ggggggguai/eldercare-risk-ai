<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="xml" encoding="UTF-8" indent="yes"/>

  <xsl:template match="@*|node()">
    <xsl:copy>
      <xsl:apply-templates select="@*|node()"/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="owner|assignee|username|email"/>

  <!-- Normalize legacy LE2I Office task names before manifest-backed conversion. -->
  <xsl:template match="task/name[starts-with(normalize-space(.), 'fall_risk_office__')]">
    <name>
      <xsl:text>fall_risk__le2i_imvia__office__</xsl:text>
      <xsl:value-of select="substring-after(normalize-space(.), '__')"/>
    </name>
  </xsl:template>
</xsl:stylesheet>
